# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-monitoring — Test Coverage Expansion v4 (April 2026)
# MAGIC
# MAGIC Runs all four coverage expansion test files to validate API alignment fixes.
# MAGIC Targets: parametric PSI/CUSUM/BAWS/sequential/calibration/pricing drift tests.

# COMMAND ----------

# MAGIC %pip install polars>=1.0 scipy>=1.12 sortedcontainers>=2.4 statsmodels>=0.14 scikit-learn pytest pytest-cov matplotlib

# COMMAND ----------

import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e",
     "/Workspace/insurance-monitoring", "--quiet"],
    capture_output=True, text=True
)
print(result.stdout[-2000:] if result.stdout else "")
print(result.stderr[-2000:] if result.stderr else "")

# COMMAND ----------

import subprocess
import sys

# Run all four coverage expansion test files
result_v4 = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        "/Workspace/insurance-monitoring/tests/test_coverage_expansion.py",
        "/Workspace/insurance-monitoring/tests/test_coverage_expansion_v2.py",
        "/Workspace/insurance-monitoring/tests/test_coverage_expansion_v3.py",
        "/Workspace/insurance-monitoring/tests/test_coverage_expansion_v4.py",
        "-v", "--tb=short", "--no-header",
    ],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-monitoring",
)
print(result_v4.stdout[-10000:])
print(result_v4.stderr[-3000:] if result_v4.stderr else "")

# COMMAND ----------

# Summary
passed = result_v4.returncode == 0

summary_lines = []
for line in result_v4.stdout.split("\n"):
    if "passed" in line or "failed" in line or "error" in line.lower():
        summary_lines.append(line)

summary = "\n".join(summary_lines) if summary_lines else "No summary lines found."
print(summary)

if not passed:
    raise RuntimeError("Coverage expansion tests FAILED — see output above.")

dbutils.notebook.exit(summary)
