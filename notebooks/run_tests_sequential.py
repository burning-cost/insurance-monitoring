# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy matplotlib sortedcontainers pytest

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys
import os

workspace_path = "/Workspace/insurance-monitoring-seq"

# Add the src directory to sys.path so insurance_monitoring is importable
sys.path.insert(0, f"{workspace_path}/src")

# Verify import works
import insurance_monitoring
print(f"insurance_monitoring loaded from: {insurance_monitoring.__file__}")
print(f"Version: {insurance_monitoring.__version__}")

from insurance_monitoring.sequential import SequentialTest, SequentialTestResult, sequential_test_from_df
print("SequentialTest, SequentialTestResult, sequential_test_from_df all importable")

# COMMAND ----------
# Run only the sequential tests
env = {
    **os.environ,
    "PYTHONPATH": f"{workspace_path}/src",
    "PYTHONDONTWRITEBYTECODE": "1",
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        f"{workspace_path}/tests/test_sequential.py",
        "-v", "--tb=long",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir={workspace_path}",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-6000:] if len(output) > 6000 else output)
