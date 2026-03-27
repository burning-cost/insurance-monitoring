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
# Run tests with coverage report
# Databricks workspace filesystem does not support __pycache__ creation.
env = {
    **os.environ,
    "PYTHONPATH": "/Workspace/insurance-monitoring/src",
    "PYTHONDONTWRITEBYTECODE": "1",
    "MPLBACKEND": "Agg",  # non-interactive matplotlib backend
}

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests",
        "-v", "--tb=short",
        "--import-mode=importlib",
        "-p", "no:cacheprovider",
        f"--rootdir=/Workspace/insurance-monitoring",
        "--cov=/Workspace/insurance-monitoring/src/insurance_monitoring",
        "--cov-report=term-missing",
        "--cov-report=term",
        "-q",
    ],
    capture_output=True,
    text=True,
    env=env,
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-8000:] if len(output) > 8000 else output)
