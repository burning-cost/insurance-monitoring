# Databricks notebook source
# COMMAND ----------
# MAGIC %pip install polars scipy pytest pytest-cov

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import subprocess
import sys

# Add the src directory to sys.path so insurance_monitoring is importable
sys.path.insert(0, "/Workspace/insurance-monitoring/src")

# Verify import works
import insurance_monitoring
print(f"insurance_monitoring loaded from: {insurance_monitoring.__file__}")
print(f"Version: {insurance_monitoring.__version__}")

# COMMAND ----------
result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests",
        "-v", "--tb=long",
        "--import-mode=importlib",
        f"--rootdir=/Workspace/insurance-monitoring",
    ],
    capture_output=True,
    text=True,
    env={**__import__("os").environ, "PYTHONPATH": "/Workspace/insurance-monitoring/src"},
)
output = result.stdout + "\n" + result.stderr
print(output)
dbutils.notebook.exit(output[-5000:] if len(output) > 5000 else output)
